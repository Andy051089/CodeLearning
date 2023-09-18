package hello;
public class TwoThree {
	public static void main(String[] args) {
		int num = 1;
		while(num<100) {
			if(num%2==0) {
				System.out.println("偶數為"+num);
			}else{
				System.out.println("基數為"+num);
			}
			if(num%3==0) {
				System.out.println("此為3的倍數"+num);
			}
			num++;
		}
	}
}
