package hello;

public class ForPrint {
	public static void main(String[] args) {
		for(int n=0;n<=200;n++) {
			if(n%7==0 && n%4!=0) {
				System.out.println(n);
			}
		}
	}

}
