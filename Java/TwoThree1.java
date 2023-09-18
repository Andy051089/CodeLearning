package hello;

public class TwoThree1 {
	public static void main(String[] args) {
		for(int num=1;num<=100;num++) {
			if(num%2==0) {
				System.out.println(num+"為偶數");
			}else {
				System.out.println(num+"為基數");
			}
			if(num%3==0) {
				System.out.println(num+"為3的倍數");
			}
		}
	}
}
